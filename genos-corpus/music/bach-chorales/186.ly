\version "2.10.33"

\header {
  title = "186 - Ach Gott, erhör' mein Seufzen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \partial 4 d4 
    f4 f4 g4 g4 
  a4 a4 bes4 a4 
  g2 f4 c4 
  f4 f4 g4 g4 
  a4 a4 g4 f4   %5
  e2 d4 a'4 
  f4 g4 e4 a4 
  f4 g4 e4 e4 
  f4 f4 g4 g4 
  a4 a4 g4 f4   %10
  e2 d4
  
}

alto = \relative c' {
  \partial 4 a4 
    d8 e8 f4 f4 e8 d8 
  c4 d4 d8 g4 f8~ 
  f8 e16 d16 e4 c4 a8 g8 
  f4 d'4 d4 c4 
  c4 f4 f8 e4 d8   %5
  d4 cis4 a4 e'4~ 
  e8 d8 d8 e8 cis4 e4 
  d8 c8 d4 c4 c4 
  c4 d4 d4 e4~ 
  e8 d16 e16 f4 e4. d8   %10
  d4 cis4 a
  
}

tenor = \relative c {
  \partial 4 f4 
    a4 c4 d4 c8 bes8 
  a8 g8 f4. e8 c'4 
  d8 g,8 c8 bes8 a4 f8 g8 
  a4 bes4~ bes8 a8 g4 
  f4 c'4 c8. bes16 a4   %5
  bes4 a8 g8 f4 a4 
  a4 bes4 a4 a4 
  a4 g4 g4 g4 
  a4 a4 bes4. c16 bes16 
  a4 d8 c8 bes4 a4   %10
  bes4 a8 g8 fis4
  
}

baixo = \relative c {
  \partial 4 d4 
    d4 a4 bes4 c4 
  f8 e8 d4 g,4 a4 
  bes4 c4 f,4 f'8 e8 
  d8 c8 bes8 a8 g8 f8 e4 
  f8 g8 a8 bes8 c4 d4   %5
  g,4 a4 d,4 cis'4 
  d4 g,4 a4 c4~ 
  c4 b4 c4 c8 bes8 
  a8 f8 d'8 c8 bes8 a8 g8 c8 
  f8 e8 d4. cis8 d4   %10
  g,4 a4 d,
  
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