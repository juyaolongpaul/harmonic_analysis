\version "2.10.33"

\header {
  title = "229 - Ich danke dir, o Gott, in deinem Throne"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \partial 4 g4 
  g4 g4 f8 ees8 d4 
  f4 g4 a4 g4 
  g4 fis4 g4 bes4 
  a4 g4 f4 d4 
  f4 g4 a4 g4   %5
  g4 fis4 g4 d'4 
  d4 d4 g,4 c4 
  c4 bes4 a4 g4 
  f2 d4 g4 
  g4 fis4 g4 fis4   %10
  g4 fis4 g4 a4 
  bes2 a4 bes4 
  a4 g4 f4 d8 ees8 
  f4 f4 g4 f4 
  ees2 d4 d'4   %15
  c4 bes4 a4 fis4 
  g4 a8 bes8 c4 bes4 
  a2 g4
  
}

alto = \relative c' {
  \partial 4 d4 
  g8 f8 ees8 d8 c4. bes8 
  d4 ees8 d8 ees4 d4 
  ees4 d4 d4 g4 
  g8 fis8 g8 ees8 c4 c8 bes16 c16 
  d8 c8 bes4 ees8 d8 e4   %5
  d4 d4 d4 g4 
  g4. fis8 g4 g4 
  d4 d4. c8 bes8 d8 
  c8 bes8 c4 bes4 d4 
  d8 ees8 d8 c8 bes8 c8 d4   %10
  d4 c4 d4 d4 
  d2 d4 d4 
  d8. c16 bes4 c4 bes4 
  a8 bes16 c16 d16 ees16 f4 ees4 d8~ 
  d8 c16 b16 c4 b4 g'4~   %15
  g8 fis8 g4. fis16 e16 d4~ 
  d8 e8 fis8 g8 a4. g8 
  g8 fis16 e16 fis4 d
  
}

tenor = \relative c' {
  \partial 4 bes8  c8 
  d4 bes4 a8 g8 f4 
  bes4 bes4 c4. bes8 
  a4~ a16 bes16 c4 bes8 d4 
  d8. c16 bes4 c8 a8 f4 
  bes8 a8 g4 c4 bes4~   %5
  bes8 a16 g16 a16 bes16 c8 bes4 bes8 c8 
  d8 c16 bes16 a8 bes16 c16 bes2 
  a4. g4 fis8 g8 bes8 
  a8 g8 a4 f4 bes8 c8 
  d8 c16 bes16 a4 g4 a4   %10
  g4 a4 bes4 a4~ 
  a4 g4 fis4 g4~ 
  g8 fis8 g8 ees8 c8 f8 f4 
  f8 g16 a16 bes4 bes4 bes8. a16 
  g2 g4 bes4   %15
  c4 d4 d4 a4 
  bes4 c8 d8 ees8 d8 d4 
  ees8 c8 a8 d8 b4
  
}

baixo = \relative c {
  \partial 4 g8  a8 
  bes4 g4 a4 bes4~ 
  bes8 a8 g4. fis8 g4 
  c4 d4 g,4 g4 
  d'4 ees4 a,8 f8 bes4~ 
  bes4 ees8 d8 c4. cis8   %5
  d2 g,4 g8 a8 
  bes8 c8 d4 ees4 e4 
  fis4 g4 d4 ees4 
  a,4 f4 bes4 g8 a8 
  bes8 c8 d4 ees4 d8 c8   %10
  bes8 c8 bes8 a8 g8 a8 g8 fis8 
  g8 a8 bes8 g8 d'4 g,4 
  d'4 ees4 a,4 bes8 c8 
  d4 d4 ees4 bes4 
  c8 d8 ees8 c8 g'4 g,4   %15
  a4 bes8 c8 d4. c8 
  bes4 a8 g8 fis4 g4 
  c4 d4 g,
  
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