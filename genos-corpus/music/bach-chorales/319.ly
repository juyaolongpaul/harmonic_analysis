\version "2.10.33"

\header {
  title = "319 - Sanctus, Sanctus Dominus Deus Sabaoth"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c'' {
  c4 a4 g2 
  f2 c'4 d4 
  c2 f4 f4 
  e4 e4 d4. c8 
  c2 c4 d4   %5
  c4 bes4 a2 
  r4 g4 a8 b8 c4 
  c4 b4 c2 
  c4 d4 bes4 bes4 
  a4 bes4 a4 g4   %10
  f2 r4 a4 
  g2 f4 c'4 
  c4 b4 c4 f4 
  f4 e4 d2 
  c2 c4 d8 c8   %15
  bes4 bes4 a4 bes8 a8 
  g2 f2 
  
}

alto = \relative c' {
  f4 f4. e16 d16 e4 
  c2 g'4 g4 
  g2 f4 g4 
  g4 e8 f16 g16 a4. g16 f16 
  e2 f4 f4   %5
  e4 d8 c8 c2 
  r4 e8 g8 f4 e4 
  a8 g16 f16 g4 g2 
  a4 a4. g16 f16 g4~ 
  g8 f16 e16 f4. e16 d16 e4   %10
  c2 r4 f4 
  f4 e4 d4 g4 
  a4 g8 f8 e4 f4 
  g4 g4 a4 g8 f8 
  e2 f4 f8 e8   %15
  d4 c4 c8 d16 e16 f4~ 
  f8 e16 d16 e4 c2 
  
}

tenor = \relative c' {
  a8 bes8 c4. g8 c8 bes8 
  a2 c4. b8 
  e2 a,4 d4~ 
  d8 c16 b16 c4. b16 a16 b4 
  g2 a4 bes8 a8   %5
  g4 f8 g8 a2 
  r4 c4 c4 c4 
  d4. e16 f16 e2 
  f4 f4 d4 c4 
  c4 bes8 d8 c4. bes8   %10
  a2 r4 c4 
  c4. bes8 a4 g4 
  d'4 d4 c4 c4 
  d4. c4 b16 a16 b4 
  g2 a4 a4   %15
  bes8 a8 g4 f8 a8 d4~ 
  d8 bes8 g8 c8 a2 
  
}

baixo = \relative c, {
  f8 g8 a8 bes8 c2 
  f,2 e'8 c8 g'8 g,8 
  c2 d8 c8 b8 g8 
  c8 e8 a8 g8 f8 d8 g8 g,8 
  c2 f4 bes,4   %5
  c4 d8 e8 f2 
  r4 c8 e8 f8 g8 a8 g8 
  f8 d8 g8 g,8 c2 
  f8 e8 d8 f8 g8 f8 e8 c8 
  f8 e8 d8 bes8 c4 c,4   %10
  f2 r4 f'4 
  c2 d4 e4 
  f4 g4 a4 a4 
  b4 c4 f,4 g4 
  c,2 f8 e8 d8 f8   %15
  g8 f8 e8 c8 f8 e8 d8 c8 
  bes8 g8 c4 f,2 
  
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