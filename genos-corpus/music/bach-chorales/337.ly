\version "2.10.33"

\header {
  title = "337 - O Gott, du frommer Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c'' {
  \partial 4 a4 
  a4 bes8 c16 d16 c4 bes16 a16 bes8 
  a4 a8 bes8 c4 f,4 
  g4 a4 g2 
  f4 a4 a4 bes8 c16 d16 
  c4 bes4 a4 a8 bes8   %5
  c4 f,4 g4 a8. bes16 
  g2 f4 g4 
  g4 a4 bes4 bes8. c16 
  a4 a4 g4 c8 d16 ees16 
  d4 c8 b8 c4 c4   %10
  c8 bes8 a4 bes4 c8 d16 ees16 
  d4 g,4 a4 a4 
  g4 g4 f4 
  
}

alto = \relative c' {
  e8 f16 g16 f4 f4 f8 e16 d16 
  e4 f4 f4 g8 e8 
  f4 e4 f4 d4 
  e4 c4 f4 f4 
  f8 g8 e8 f4 e8 f4   %5
  f4 f8 ees8 d4 e4 
  f4 f4 e4 c4 
  d4 e4 f4. e16 d16 e4 
  f4 ees4 d4 g4 
  f8 aes8 g8 f8 e4 g  %10
  f4 f4 f4 
  f4 f4 e4 c4 f4~
  f4 e4 c4 

}

tenor = \relative c' {
  \partial 4 cis4 
  d4 d8 bes8 a8 g16 f16 g4 
  c4 c8 d8 e4 a,8 f8 
  c'4 c4 bes4 g4 
  a4 c4 d4 d8 bes8~ 
  bes8 a8 g8 c8 c4 f,8 g8   %5
  a8 g8 f4 c'4 c4 
  c8 bes16 a16 bes8 g8 a4 d8 b8 
  c4 c4 d8 c16 bes16 c4 
  c4 c4 b4 c4 
  c8 f8 d4 c4 e4   %10
  c4 c4 d4 c4 
  bes4 bes4 a4 d4 
  d8 g,8 a8 bes8 a4 
  
}

baixo = \relative c {
  \partial 4 a4 
  d8 c8 bes4 c4 c4 
  f,4 f'4 e4 d4 
  c8 bes8 a8 f8 bes8 g8 c8 c,8 
  f4 f'8 e8 d8 c8 bes8 g8 
  c4 c4 f,4 d'4   %5
  a4 bes4 bes'8 a8 g8 f8 
  c4 c4 f4 b,8 g8 
  c4 bes8 a8 g4 c4 
  f4 fis4 f4 ees8 f16 g16 
  aes8 g16 f16 g8 g,8 c4 c'8 bes8   %10
  a8 g8 f8 ees8 d8 c8 bes8 a8 
  bes4 c4 f8 e8 d8 c8 
  bes8 b8 c4 f4 
  
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