\version "2.10.33"

\header {
  title = "223 - Ich dank' dir, Gott für all' Wohltat"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \partial 4 g4 
  e4 fis4 g4 g4 
  e4 d4 c4 c4 
  g'4 g4 a4 b4 
  c4 b4 a4 a4 
  g2 r4 g4   %5
  a4 b4 c4 g4 
  a4 g4 f4 f4 
  e2 r4 c'4 
  a4 b4 c4 a4 
  g4 f4 e4 g4   %10
  e4 fis4 g4 c4 
  b4 a4 g4 g4 
  e8 f8 g4 a4 g4 
  f4 e4 d4 d4 
  c1   %15
  
}

alto = \relative c' {
  \partial 4 c4 
  c4 c4 b4 d4~ 
  d8 c4 b8 g4 g4 
  d'4 e4. fis8 g4~ 
  g8 fis8 g2~ g8 fis8 
  d2 r4 e4~   %5
  e8 d4 e8 e4 c4~ 
  c8 b8 c8 cis8 d8 cis8 d4~ 
  d4 cis4 r4 g'4~ 
  g8 f16 e16 f4 e4 f4 
  e4. d4 cis8 d4   %10
  e4 b4 b4 g'8 a8~ 
  a8 g4 fis8 d4 d4 
  c4 c4 c8 f4 e8 
  d4. c4 b16 a16 b4 
  g1   %15
  
}

tenor = \relative c {
  \partial 4 e8  f8 
  g4 a4 d,4 g4 
  g4~ g8. f16 e4 e4 
  g4 c4 c4 d4 
  c4 d4 e4 d8. c16 
  b2 r4 c4   %5
  c4 b4 a4 e4 
  f4 g4 a4 a4 
  a2 r4 c4 
  c4 d4 g,4 d'8 c8 
  bes4 a4 a4 g4~   %10
  g8 a4 g16 fis16 e4 e'8 d8 
  d4 d8. c16 b4 g4 
  g4 c8 bes8 a8. b16 c8. bes16 
  a8 g8 g4 a8 d,8 g8 f8 
  e1   %15
  
}

baixo = \relative c {
  \partial 4 c4 
  bes4 a4 g4 b4 
  c4 g4 c,4 c'4 
  b4 c8 b8 a4 g4 
  e'4 d4 c4 d4 
  g,2 r4 e'4   %5
  f4 gis,4 a4 c4 
  f4. e8 d8 e8 f8 d8 
  a'2 r4 e4 
  f8 e8 d4 c4 d4~ 
  d8 cis8 d4 a4 b4   %10
  cis4 dis4 e4 e8 fis8 
  g4 d4 g,4 b4 
  c8 d8 e8 c8 f4 c4~ 
  c8 b8 c4 f,4 g4 
  c,1   %15
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}